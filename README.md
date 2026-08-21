# NestyNet\_SR: Symbolic Regression and Differential Equation Discovery

**NestyNet\_SR** is a Python package for discovering analytical expressions and differential equations from data using neural network surrogates. Built on top of [NestyNet](https://github.com/RodrigoIbata/NestyNet), it exploits accurate analytic derivatives for reliable separability detection and structure discovery.

This README is the consolidated project guide.

---
FLOW:

```text
main()  [nestynet_sr/run_SR.py]
├─ parse args + load CSV(s) + build y-transform registry
├─ blinded-mode guard (--blinded forbids answer-key access during the run)
├─ statistical audit firewall: search-vs-audit data split (--stat_* flags)
├─ generalized-symmetry Stage-A layer (default ON; --gs-no-stagea to disable)
├─ optional committee-of-experts scout proposers (--coe_* flags)
├─ internal phase prescan (automatic; skipped with --fast / --resume_from)
│
├─ Stage A
│  ├─ identity baseline fit (with optional asinh/log fallback paths)
│  ├─ optional prescan winner tried before identity separability
│  ├─ fallback identity separability + quick/full passes over promising transforms
│  ├─ optional outer-peel transform ranking (and optional square autorun)
│  └─ save checkpoint: results/<stem>.state.pkl (phase="after_stageA")
│
├─ Stage B (default on unless --no_stageB; needs a Stage-A model/AST)
│  ├─ pick Stage-B y-space (Stage-A choice, plus ranked proposal fallback logic)
│  ├─ A <-> B feedback loop (max_ab_iters; default 5, forced to 1 for multi-dataset)
│  │  ├─ run Stage B: run_stageB_from_model(...)
│  │  ├─ optional identity fallback if ranked y-transform underfits
│  │  └─ iteration > 1: convergence checks + Stage-A re-run on Stage-B AST
│  │     (run_separability_for_transform with freeze_non_nn=True, skip_initial_fit=True)
│  └─ save results/<stem>_stageB.pkl and results/<stem>_final.human
│
├─ optional Class-SR joint fitting (+ Parameter-SR invariants) for multi-dataset runs
├─ optional first-class DE discovery (--discover_de)
├─ generalized-symmetry reports: results/<stem>.gs_report.{json,md}
├─ committee-of-experts exit audits and final committee (when enabled)
├─ final Pareto polish -> results/<stem>_polish/
├─ statistical-selection certificate (--stat_certificate_json, --stat_archive_json)
└─ write JSON report: results/<stem>.report.json
```


## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Tutorial: End-to-End Symbolic Regression](#tutorial-end-to-end-symbolic-regression)
5. [Pipeline Overview](#pipeline-overview)
6. [Class-SR (Multi-Dataset Shared Constants)](#class-sr-multi-dataset-shared-constants)
7. [DE Discovery (Detailed)](#de-discovery-detailed)
8. [SR-First DE Discovery](#sr-first-de-discovery)
9. [Multi-Dataset DE Discovery](#multi-dataset-de-discovery)
10. [VarPro Refinement](#varpro-refinement)
11. [Complex Number Support](#complex-number-support)
12. [PDE Extension (Roadmap)](#pde-extension-roadmap)
13. [Dimensional Analysis](#dimensional-analysis)
14. [Generalized Symmetries](#generalized-symmetries)
15. [Statistical Selection, Blinded Mode & Final Polish](#statistical-selection-blinded-mode--final-polish)
16. [Worked Examples & Paper Vignettes](#worked-examples--paper-vignettes)
17. [Directory Structure](#directory-structure)
18. [Output Files](#output-files)
19. [Testing](#testing)
20. [License & Citation](#license--citation)
21. [References](#references)

---

## Features

- **Symbolic Regression**: Discover algebraic expressions `y = f(x₁, x₂, ...)` from data
- **Class-SR**: Multi-dataset symbolic regression with shared ("class") vs per-dataset ("experiment") constants
- **DE Discovery**: Discover 1D implicit DEs from trajectory data via STLSQ + optional VarPro / Stage-B residual refinement
- **SR-First DE Output**: Run DE discovery as a first-class output of `run_SR.py` via `--discover_de`
- **System / Vector DE Discovery**: Coupled multi-equation systems (Maxwell, Navier-Stokes) with coefficient tying and vector-calculus macros
- **Hamiltonian Discovery**: Phase-space trajectory analysis preserving symplectic structure
- **Separability Detection**: Additive, multiplicative, and compound-variable structure via mixed partial derivatives
- **Stage B Rewrites**: 40+ rules transforming neural atoms into analytical forms (polynomials, sinusoids, rationals)
- **Generalized Symmetries**: default-on Stage-A coordinate discovery via a learned general-affine determining operator, with charts, a recursive carrier bank, and noise-calibrated promotion
- **Statistical Selection**: certified model selection behind a search-vs-audit data firewall, with a blinded benchmark mode
- **Final Pareto Polish**: post-run accuracy-complexity polish written to `results/<stem>_polish/`
- **factorized symbolic search**: Brute-enumerate + UCB-guided mutation engine for expression tree search
- **Variable Projection (VarPro)**: Two-phase optimization for ~100× coefficient accuracy improvement
- **Dimensional Analysis**: Local consistency checks plus global constraint propagation (Buckingham-Sudoku)
- **Complex DE Support**: 2-component decomposition with coefficient-tied discovery and physics-notation output
- **GUI Application**: Interactive Streamlit interface (optional)

---

## Installation

### Prerequisites

- Python ≥ 3.10
- PyTorch ≥ 2.6
- NumPy ≥ 2.2
- SciPy ≥ 1.15
- pandas ≥ 2.2
- NestyNet ≥ 0.1.0
- matplotlib ≥ 3.5
- sympy ≥ 1.12

Optional:
- streamlit, altair (GUI; `pip install -e ".[gui]"`)
- sphinx (documentation; `pip install -e ".[docs]"`)

Console scripts installed with the package: `nestynet-sr`, `nestynet-de`, and
`nestynet-polish` (standalone final Pareto polish over an existing run).

### From PyPI (once published)

```bash
pip install nestynet-sr
```

### From Source

```bash
# Install NestyNet first (required dependency)
cd /path/to/NestyNet
pip install -e .

# Then install NestyNet_SR
cd /path/to/NestyNet_SR
pip install -e .

# Optional extras
pip install -e ".[gui]"   # Streamlit GUI
pip install -e ".[dev]"   # pytest, ruff, mypy
```

To recreate the four AI Feynman workspaces used for Paper III from the
separately archived Zenodo data, see
[`PAPER3_REPRODUCIBILITY.md`](PAPER3_REPRODUCIBILITY.md).
The reproducible truth-blind cheap-to-CoE controller is documented in
[`CAMPAIGN_ESCALATION.md`](CAMPAIGN_ESCALATION.md).

Paper IV uses the DE path in this same repository. Its exact scalar, complex,
Maxwell and 308-asteroid inputs plus compact reference artifacts are handled by
the separate workflow in
[`PAPER4_REPRODUCIBILITY.md`](PAPER4_REPRODUCIBILITY.md).

---

## Quick Start

### Symbolic Regression

```bash
# Basic symbolic regression
nestynet-sr --filepath data/my_data.csv

# Fast mode for testing
nestynet-sr --filepath data/my_data.csv --fast

# With dimensional analysis (on by default when units are provided)
nestynet-sr --filepath data/my_data.csv \
    --y_units "[1,0,0]" --x_units "[[1,0,0],[0,1,0]]"
```

### DE Discovery

```bash
# Basic DE discovery
nestynet-de --filepath data/my_ode_data.csv

# With VarPro refinement
nestynet-de --filepath data/my_ode_data.csv --varpro

# 2nd-order DE with template search
nestynet-de --filepath data/my_ode_data.csv \
    --order_candidates 2 --include_xdu --varpro --varpro_templates power,exp
```

### Multi-Dataset DE Discovery

```bash
nestynet-de --filepaths data/exp1.csv data/exp2.csv data/exp3.csv --varpro
```

### Class-SR (Multi-Dataset SR)

```bash
nestynet-sr --filepaths examples/classSR/data/quad_1.csv \
    examples/classSR/data/quad_2.csv examples/classSR/data/quad_3.csv \
    --class_sr --factorized-search
```

### SR + First-Class DE Output

```bash
nestynet-sr --filepath data/dho_sr.csv --discover_de \
    --de_order_candidates 2 --de_include_du --de_no_x --de_no_xu --de_no_xdu
```

### Streamlit GUI

```bash
streamlit run nestynet_sr/gui_sr.py
```

Oracle continuous skeleton refinement lab GUI:

```bash
streamlit run examples/oracle_factorized_search/oracle_lab_streamlit.py
```

See `examples/oracle_factorized_search/README.md` for oracle lab CLI/GUI details.

---

## Tutorial: End-to-End Symbolic Regression

This walkthrough shows how to generate synthetic data for a known equation, run the SR pipeline to rediscover it, and inspect the results — first without units, then with dimensional analysis enabled.

### Step 1: Generate Data

Use `data/generate_data.py` to create a CSV from any expression using SymPy syntax. Variables must be named `x0`, `x1`, etc.

```bash
# Example: y = sin(x0 * x1) + x2^2
python data/generate_data.py \
    --expr "sin(x0 * x1) + x2**2" \
    --output data/tutorial.csv \
    --samples 10000 \
    --min 1.0 --max 5.0
```

This produces `data/tutorial.csv` with columns `y, x0, x1, x2`.

You can also set per-variable ranges:

```bash
python data/generate_data.py \
    --expr "sin(x0 * x1) + x2**2" \
    --output data/tutorial.csv \
    --xmin "[0.5, 0.5, 1.0]" --xmax "[3.0, 3.0, 5.0]"
```

### Step 2: Run Symbolic Regression (Without Units)

```bash
python nestynet_sr/run_SR.py --filepath data/tutorial.csv
```

This runs the full pipeline:
1. **Stage A** — trains a neural network surrogate and detects separability structure (additive/multiplicative splits, compound variables)
2. **Stage B** — rewrites surviving neural atoms into analytical forms (polynomials, sinusoids, rationals, etc.)

For faster iteration during testing, use `--fast`:

```bash
python nestynet_sr/run_SR.py --filepath data/tutorial.csv --fast
```

### Step 3: Inspect the Results

Results are written to `results/`:

| File | Content |
|------|---------|
| `results/tutorial.human` | Human-readable discovered expression |
| `results/tutorial.state.pkl` | Checkpoint (can resume with `--resume_from`) |
| `results/tutorial.report.json` | Detailed JSON report with metrics |

View the discovered expression:

```bash
cat results/tutorial.human
```

### Step 4: Run with Dimensional Analysis

To constrain the search with physical units, provide a unit exponent vector for `y` and a unit exponent matrix for the input variables. Each vector lists exponents in a chosen basis (e.g., Length, Time, Mass).

For example, suppose `x0` has units of length [L], `x1` has units of inverse length [L⁻¹] (so that `x0 * x1` is dimensionless, as required for `sin`), `x2` has units of length [L], and `y` has units of length² [L²]:

```bash
python nestynet_sr/run_SR.py --filepath data/tutorial.csv \
    --y_units "[2,0,0]" \
    --x_units "[[1,0,0],[-1,0,0],[1,0,0]]" \
    --units_basis "L,T,M"
```

The three entries in each vector correspond to the exponents of [L, T, M]. When a units spec is provided, dimensional analysis is enabled by default (pass `--ignore_units` to disable). This activates:
- **Local consistency**: addition requires matching dimensions; transcendentals require dimensionless arguments
- **Global constraint propagation**: all dimensional constraints are solved jointly to prune infeasible candidates early

### Step 5: Resume from Checkpoint

If you want to continue a previous run (for instance to extend Stage B):

```bash
python nestynet_sr/run_SR.py --filepath data/tutorial.csv --resume_from results/tutorial.state.pkl
```

### Summary of Commands

| Step | Command |
|------|---------|
| Generate data | `python data/generate_data.py --expr "sin(x0*x1)+x2**2" --output data/tutorial.csv` |
| Run SR (no units) | `python nestynet_sr/run_SR.py --filepath data/tutorial.csv` |
| Run SR (fast) | `python nestynet_sr/run_SR.py --filepath data/tutorial.csv --fast` |
| Run SR (with units) | `python nestynet_sr/run_SR.py --filepath data/tutorial.csv --y_units "[2,0,0]" --x_units "[[1,0,0],[-1,0,0],[1,0,0]]"` |
| View result | `cat results/tutorial.human` |
| Resume | `python nestynet_sr/run_SR.py --filepath data/tutorial.csv --resume_from results/tutorial.state.pkl` |

---

## Pipeline Overview

### Symbolic Regression (`nestynet-sr`)

| Stage | What it does |
|-------|-------------|
| **Stage A** | Train NN surrogate, detect separability (additive/multiplicative/compound) via mixed partial derivatives |
| **Stage B** | Rewrite neural atoms into analytical forms; factorized symbolic search explorer runs first, then specialized rules |
| **Class-SR (optional, multi-dataset)** | Auto-classify leaf tags as shared vs per-dataset, then run a joint fit with tied parameters |
| **First-class DE (optional)** | Discover DE residuals from SR surrogates (`--discover_de`) and optionally run Stage-B on the residual |
| **Y-transforms** | Automatically try 15 output transformations (identity, square, log, reciprocal, …) |
| **Outer-peel** | Suggest non-identity transforms when identity fails |

**Separability types detected**:
- **Additive**: `f(x,y) = g(x) + h(y)` — via `∂²f/∂x∂y ≈ 0`
- **Multiplicative**: `f(x,y) = g(x) · h(y)` — via `∂²log|f|/∂x∂y ≈ 0`
- **Compound variables**: `z = ∏ xᵢ^αᵢ` — via rank-1 Jacobian in log-space
- **Generalized compounds**: linear, radial, and translation-centered coordinates

### DE Discovery (`nestynet-de`)

| Step | What it does |
|------|-------------|
| 1. Train surrogate | Fit segmented NN `u(x)` to trajectory data |
| 2. Compute derivatives | Analytic `u_x`, `u_xx`, … from NestyNet |
| 3. Build library | Candidate terms: polynomials, products, derivatives |
| 4. STLSQ | Sequential Thresholded Least Squares for sparse selection |
| 5. VarPro Phase 1 | Refine linear coefficients |
| 6. VarPro Phase 2 | Search over nonlinear templates (power laws, exponentials) |

### Key Differences

| Feature | `nestynet-sr` | `nestynet-de` |
|---------|---------------|----------------|
| **Discovers** | Algebraic expressions `y = f(x)` | Differential equations `u_x + g(x,u) = 0` |
| **Method** | Separability + Stage B rewrites | STLSQ on surrogate derivatives |
| **Output** | Explicit function | Implicit DE |
| **Phases** | Stage A (NN) + Stage B (rewrites) | STLSQ + VarPro Phase 1 & 2 |

---

## Class-SR (Multi-Dataset Shared Constants)

`Class-SR` runs inside `run_SR.py` after Stage B when using `--filepaths ... --class_sr`.

### What it does

1. Starts from per-dataset Stage-B fits of the same expression structure.
2. Auto-classifies leaf tags as class-shared vs experiment-specific using cross-dataset coefficient variation (CV).
3. Runs a joint optimization with shared parameters tied across datasets.

### Current auto-classification logic

- Uses `--class_cv_threshold` (default `0.15`) on per-tag CV.
- By default, auto-classification focuses on free-constant leaves and excludes pure scale leaves.
- If strict filtering finds no class tags, it retries once with relaxed inclusion (includes scale + non-free leaves).
- Explicit free-constant scopes are respected (`scope=class` and `scope=experiment` from declared constants).
- A greedy inclusion pass plus safety fallback prevents severe validation-loss regressions from over-sharing.

### Parameter-SR (derived invariants)

Parameter-SR is enabled by default in Class-SR and can discover derived invariants that act as soft constraints during joint fitting.

- Disable with `--no_class_param_sr`
- Metadata-assisted invariants via `--class_param_sr_metadata` (row-wise list, column-wise dict, or dataset-keyed dict)
- Key controls: `--class_param_sr_max_invariants`, `--class_param_sr_score_threshold`, `--class_param_sr_penalty_weight`

### Optimizer backend

- `--class_sr_optimizer lbfgs` (default)
- `--class_sr_optimizer lm_tie` (LM with explicit tie constraints)

### Outputs

- `results/<stem>_classSR.json`
- `results/<stem>_classSR.human`

---

## DE Discovery (Detailed)

Current CLI implementation is focused on 1D DEs (`u(x)` or `u(t)`) using a selected axis (`--x_axis`), while the derivative stack can already evaluate multi-axis terms for future PDE extensions.

### Algorithm

1. **Train Surrogate**: Fit a segmented neural network `u(x)` to the data
2. **Compute Derivatives**: Use analytic derivatives to get `u_x`, `u_xx`, etc.
3. **Build Library**: Construct candidate terms `φ_k(x, u, u_x, ...)`
4. **STLSQ Selection**: Sparse regression `u̇ = Θ(u,x) ξ` with iterative thresholding
5. **Unbiased Refit**: Ridge-free refit on selected support (removes shrinkage bias)
6. **Return DE**: Implicit form `anchor + Σ c_k φ_k = 0`

### Operator-Factorized FSS Mode

For paper-facing STLSQ-free DE discovery, use the operator-factorized first-line
mode.  It builds typed residual-ratio lanes such as `u_x + a(x)u = 0`,
`u_x + b(u) = 0`, `u_xx + f(u) = 0`, and two-block assemblies before falling
back to broad whole-RHS FSS.

```bash
python nestynet_sr/run_de.py \
  --filepath data/example.csv \
  --order_candidates 1,2 \
  --factorized-de \
  --factorized-de-whole-rhs auto \
  --factorized-search-budget-scope global \
  --de-coe-mode adjudicate \
  --save_json
```

Recommended controls:

- `--factorized-de-whole-rhs never|auto|always`: `auto` keeps broad FSS as
  a bounded reservoir source only when typed evidence is absent or ambiguous;
  `always` preserves the legacy broad-FSS ablation path.
- `--factorized-search-max-attempts N`: caps broad whole-RHS heuristic attempts.
  In `--factorized-de` auto mode, the default effective cap is `1`.
- `--factorized-search-integrate-topk 0`: disables broad-FSS internal rollout
  when benchmark or DE-CoE rollout arbitration is responsible for validation.
- `--factorized-search-budget-scope global`: divides broad-FSS fit/probe row
  budgets across trajectories instead of spending the full budget per group.

The typed lane metadata in JSON reports includes lane, carrier, coordinate,
family, collapse evidence, family-gate counters, explorer launches, scheduler
skips, and two-block assembly counters.  Least-squares polish inside a typed
slot is calibration of that slot, not a fixed STLSQ library prior.

For the broad whole-RHS FSS ablation/rescue route, use:

```bash
python nestynet_sr/run_de.py \
  --filepath data/example.csv \
  --factorized-search-only \
  --factorized-search-budget-scope per_group \
  --save_json
```

### Term Library

For `nestynet-de`, the default RHS library is intentionally small:

- base terms: `1`, `x`, `u`, and `x·u` (with powers controlled by `--max_x_power`, `--max_u_power`)
- anchor term on the left: `u_x` (order 1) or `u_xx` (order 2)

Optional terms are off by default and can be enabled explicitly:

- `--include_xdu` for `x·u_x` (important for equations like Lane-Emden)
- `--include_udu` for `u·u_x`
- `--include_du` for `u_x` in the RHS library when not used as anchor
- `--include_d2u` for `u_xx` in the RHS library when not used as anchor

Note: in `run_SR.py --discover_de`, defaults are slightly richer (`--de_max_u_power 2` and `x·u_x` included unless `--de_no_xdu`).

### CLI Reference

**Required Inputs** (choose one):
- `--filepath`: Single CSV file (single-dataset mode)
- `--filepaths`: Multiple CSV files (multi-dataset mode)

**DE Configuration**:
- `--order_candidates 1,2`: Comma-separated DE orders to try
- `--x_axis 0`: Index of independent variable (auto-detected from coordinate metadata when available)
- `--max_x_power 1`: Maximum power of x
- `--max_u_power 1`: Maximum power of u

**Term Library**:
- `--include_xdu`: Include `x·du` terms (critical for 2nd-order DEs like Lane-Emden)
- `--include_udu`: Include `u·du` terms (logistic-like dynamics)
- `--include_du`: Include `du` in library
- `--include_d2u`: Include `d2u` in library

**STLSQ Parameters**:
- `--stlsq_lambda 0.001`: Sparsification threshold (larger → sparser)
- `--stlsq_max_iter 10`: Maximum iterations
- `--ridge 1e-10`: Ridge regularization during STLSQ

**Surrogate Training**:
- `--num_segments 32`: NN segments
- `--epochs 8000`: Maximum epochs
- `--loss_target 1e-7`: Target loss
- `--strategy direct_solve`: LM strategy

**Output**:
- `--output_dir results`: Output directory
- `--save_json`: Save JSON report

**Residual Stage-B Refinement**:
- `--stageb_refine_residual`: Run Stage-B LM fit on discovered residual AST(s) with zero targets
- `--stageb_epochs 500`: Epoch cap for this residual refinement

### Example: Gaussian PDF (AIF #000)

**Ground Truth**: `u(x) = (1/√2π) exp(-x²/2)`

**1st-Order Discovery**:
```
u_x0 + x0 * u = 0
coefficient: 0.999995 (error: 5e-6)
RMS: 5e-6
```

**2nd-Order Discovery**:
```
u_x0x0 + u + x0 * u_x0 = 0
coefficients: 1.00094, 1.00033 (errors: ~1e-3)
```

### Understanding RMS Values

- **RMS ~1e-5 to 1e-6**: Excellent (derivative consistency matches surrogate precision)
- **RMS ~1e-4 to 1e-5**: Good (typical for 2nd-order DEs)
- **RMS >1e-3**: May indicate wrong equation or insufficient surrogate quality

Second derivatives compound numerical errors — the limiting factor is often derivative consistency rather than coefficient error.

### Tips

1. **For 2nd-order DEs**: Always use `--include_xdu` if you expect `x·u_x` terms
2. **Enforce sparsity**: Increase `--stlsq_lambda` (try 0.1) to avoid spurious terms
3. **Train well**: Target surrogate loss ~1e-12 or better for clean DEs
4. **Compare orders**: Lower RMS doesn't always mean the correct equation

---

## SR-First DE Discovery

`run_SR.py` can emit DE discovery as a first-class SR output without leaving the SR workflow.

### Workflow

1. Train SR surrogate(s) as usual (single or multi-dataset).
2. Discover DE residual(s) from surrogate derivatives.
3. Optionally promote low-CV coefficients to shared Class-DE constants in multi-dataset mode.
4. Optionally run Stage-B on the DE residual against zero targets.

### Key Flags

- `--discover_de`: enable DE discovery in `run_SR.py`
- `--de_y_space identity|final`: discover in raw `y` space or final Stage-A transformed space
- `--de_order_candidates`, `--de_x_axis`, `--de_max_x_power`, `--de_max_u_power`
- library controls: `--de_no_const`, `--de_no_x`, `--de_no_u`, `--de_no_xu`, `--de_no_xdu`, `--de_include_du`, `--de_include_d2u`, `--de_include_udu`
- `--de_class_de`, `--de_class_de_cv`: class-share stable DE coefficients across datasets
- `--no_de_stageB`, `--de_stageB_max_outer_iters`, `--de_stageB_epochs`: residual Stage-B controls

### Artifacts

- `results/<stem>_de.pkl` (rich payload: config, term ASTs, coefficients, residual ASTs)
- `results/<stem>_de.human`
- `de` block embedded in `results/<stem>.report.json`

---

## Multi-Dataset DE Discovery

Multi-dataset mode discovers DEs with **shared term support** across multiple datasets while allowing **dataset-specific coefficients**. Use this when experiments share the same physical law but have different parameters.

### When to Use Multi-Dataset Mode

- Same underlying law with different parameters (for example logistic growth with varying `r`)
- Identifying universal terms vs. calibration-specific coefficients
- Improving identifiability with multiple trajectories under different conditions
- Relating discovered coefficients to known metadata (temperature, concentration, forcing, etc.)

### Usage

```bash
# Multiple CSV files
nestynet-de --filepaths data/exp1.csv data/exp2.csv data/exp3.csv

# With wildcards
nestynet-de --filepaths data/logistic_*.csv --order_candidates 1 --max_u_power 2
```

### Algorithm: Group-Sparse STLSQ

1. **Train separate surrogates**: One NN `u_d(x)` per dataset
2. **Build shared library**: Same feature matrix `Φ` for all datasets
3. **Group-sparse regression**:
   - Stack per-dataset coefficients into matrix `C ∈ ℝ^(D×K)`
   - Compute row-wise L2 norms `‖C[k,:]‖₂` for each term `k`
   - Threshold: keep term `k` if `‖C[k,:]‖₂ ≥ λ` (group sparsity)
   - Re-solve per-dataset coefficients on selected support
4. **Unbiased refit**: Ridge-free refit per dataset
5. **Model selection**: Choose order minimizing `mean(RMS_val) + penalty·num_terms`

**Key Idea**: Terms are either active in **all datasets** or **none**. Coefficients vary per dataset.

### Example: Logistic Growth

```
Ground Truth: u' = r·u·(1 - u/K)  ⟹  u' = r·u - (r/K)·u²

Dataset A: r=0.5, K=10  ⟹  u' = 0.5·u - 0.05·u²
Dataset B: r=0.8, K=10  ⟹  u' = 0.8·u - 0.08·u²
Dataset C: r=1.0, K=10  ⟹  u' = 1.0·u - 0.10·u²

Discovered: Same support {u, u²} with dataset-specific coefficients
```

### Output Format

Each dataset gets its own equation with the **same active terms** but **different coefficients**:
```
Discovered terms (shared support):
  term[0]: U()
  term[1]: Pow(U(), 2)

Dataset 0: u_x0 + 0.5*u + -0.05*u^2 = 0  (RMS: 1.23e-06)
Dataset 1: u_x0 + 0.8*u + -0.08*u^2 = 0  (RMS: 1.45e-06)
Dataset 2: u_x0 + 1.0*u + -0.10*u^2 = 0  (RMS: 1.67e-06)
```

### Validation Workflow

1. Check whether shared support matches expected physics.
2. Compare per-dataset coefficient patterns against known parameter trends.
3. Hold out one dataset and validate the discovered structure out-of-sample.
4. Inspect residual consistency; outlier RMS often indicates data or model mismatch.

### Limitations

- All datasets must have the same number of independent variables.
- Group sparsity enforces globally shared support by design; if regimes truly differ, split the cohort first.

---

## VarPro Refinement

### Phase 1: Linear Coefficient Refinement

Analytically optimizes linear coefficients on the full training batch (no mini-batch bias):

```bash
nestynet-de --filepaths data/exp1.csv data/exp2.csv --varpro
```

Typically reduces training RMS by 10–50% vs STLSQ alone.

### Phase 2: Nonlinear Template Search

Discovers nonlinear terms where **shape parameters ψ are shared** across datasets but **linear coefficients β vary**:

```bash
nestynet-de --filepaths data/exp1.csv data/exp2.csv \
    --varpro_templates power,exp --template_lm
```

**Template families** (via `--varpro_templates`):
- `power`: Power laws `u^p`, `x^p`
- `exp`: Exponentials `exp(k·u)`, `exp(k·x)`

Currently implemented and production-tested families are `power` and `exp`.

**How it works**:
1. Generate candidate template instances per family
2. Initialize nonlinear parameters ψ heuristically
3. If `--template_lm`: optimize ψ jointly across all datasets via LM
4. VarPro analytically eliminates per-dataset linear coefficients
5. Apply group-sparse STLSQ on extended library (baseline + template)
6. Select best template (or baseline if none improve)

### Advanced Options

```bash
nestynet-de --filepaths data/*.csv \
    --varpro_templates power,exp \
    --template_lm \
    --template_lm_epochs 500 \
    --support_minimization \
    --rms_tol_factor 1.05 \
    --max_templates 3 \
    --complexity_penalty 0.001 \
    --prefer_autonomous \
    --prefer_forced
# --prefer_autonomous penalizes x-dependent terms;
# --prefer_forced penalizes state-only terms.
```

Datasets are weighted by sample count during shared-parameter template fitting so large datasets do not dominate via repeated mini-batches.

---

## Complex Number Support

Complex fields are handled via a **2-component real decomposition** with optional coefficient-tied discovery.

### 2-Component Approach

Complex fields are represented as 2-output real systems:

```
ψ = u + iv  →  [u, v] (2-output surrogate)
```

A complex PDE like the Schrödinger equation `i∂ψ/∂t = -∂²ψ/∂x²` decomposes into:

```
∂u/∂t = -∂²v/∂x²
∂v/∂t = +∂²u/∂x²
```

The existing `discover_system_de_from_surrogate` handles this directly.

Implemented reference tests:

| ID | Equation | Description | Status |
|----|----------|-------------|--------|
| CDE000 | `dz/dt = i·ω·z` | Rotating phasor | ✅ Passing |
| CDE010 | `i∂ψ/∂t = -∂²ψ/∂x²` | Free Schrödinger | ✅ Passing |

| Physics Domain | Supported | Notes |
|----------------|-----------|-------|
| Free Schrödinger | ✅ | Linear, clean cross-coupling |
| Damped oscillators | ✅ | Linear complex DE |
| Diffusion (complex D) | ✅ | Linear |
| Maxwell equations | ✅ | Naturally real |
| NLS (nonlinear Schrödinger) | ✅ | Requires `extra_terms` or `complex_ops` |

### complex_ops Module

The `nestynet_sr.sr_de.complex_ops` module provides higher-level helpers:

```python
from nestynet_sr.sr_de import (
    Psi, AbsSqPsi, DPsi, D2Psi,
    ComplexDESearchConfig, make_laplacian_term, make_nls_term,
    discover_complex_de_from_surrogate,
)

# Library helpers
psi = Psi()
nls_u, nls_v = AbsSqPsi(psi)            # |ψ|²ψ in one line
du_dx, dv_dx = DPsi(axis=1, z=psi)      # First derivatives
d2u_dx2, d2v_dx2 = D2Psi(1, 1, z=psi)  # Second derivatives

# Coefficient-tied discovery
cfg = ComplexDESearchConfig(
    complex_terms=[make_laplacian_term(psi), make_nls_term(psi)],
)
result = discover_complex_de_from_surrogate(surrogate, loader, cfg=cfg)

# Physics notation output
print(result.format_complex_equation())
# → "i dpsi/dt = -d2psi/dx2 + 0.5*|psi|^2*psi"
print(result.is_valid_complex)  # True if coefficient structure is valid
```

### Limitations of the Raw 2-Component Workflow

1. Library construction is verbose for nonlinear terms (for example `|psi|^2 psi`).
2. Unconstrained system discovery can produce coefficient asymmetry between real/imaginary equations.
3. Output is a coupled real system unless formatted back into complex physics notation.

### AST Complex Nodes

Five complex-valued AST nodes are available:

| Node | Operation | Definition |
|------|-----------|-----------|
| `ConjNode` | Complex conjugate | `conj(z) = x - iy` |
| `RealNode` | Real part | `real(z) = x` |
| `ImagNode` | Imaginary part | `imag(z) = y` |
| `AbsNode` | Modulus | `abs(z) = √(x² + y²)` |
| `ArgNode` | Phase angle | `arg(z) = atan2(y, x)` |

All support forward evaluation, JVP, VJP, and Hessians. `ConstNode` also supports complex scalars.

Implementation status:

| Component | Status | Notes |
|-----------|--------|-------|
| Node definitions | ✅ | `sr_core/bridges.py` |
| Forward/JVP/VJP/Hessian | ✅ | Propagates through complex nodes |
| Complex leaf optimization | 🟡 | Real-valued NestyNet leaves limit full native path |
| Direct complex AST DE wiring | 🟡 | Prefer `complex_ops` for end-to-end workflows |

### Barriers to Native Complex in NestyNet

Full native complex128 support in NestyNet's LM optimizer would require:

| Issue | Severity | Description |
|-------|----------|-------------|
| `torch.finfo()` | 🔴 Crash | Fails on complex dtypes (~15 locations) |
| `J^T J` Hessian | 🔴 Wrong math | Should be `J^H J` for complex |
| `torch.sign()` | 🔴 Undefined | No sign for complex numbers |
| `torch.isfinite()` | 🔴 Fails | Doesn't work on complex tensors |
| Robust loss | 🟡 | `.abs()` breaks complex gradients |
| Constraint bounds | 🟡 | ±∞ bounds undefined for complex |

**Recommendation**: Use the 2-component approach (works now) or the `complex_ops` helpers (coefficient-tied, physics-notation output). Native complex is a separate NestyNet-core project.

### Upgrade Path Options

1. Stay with 2-component decomposition (already production-usable).
2. Use `complex_ops` helpers (`Psi`, `AbsSqPsi`, `DPsi`, `D2Psi`) for cleaner term construction.
3. Use constrained complex discovery (`discover_complex_de_from_surrogate`) for coefficient tying and physics-notation output.
4. Implement native complex support in NestyNet core if end-to-end complex LM optimization becomes a priority.

---

## PDE Extension (Roadmap)

The current CLI targets 1D DE discovery (`u(x)` or `u(t)`), but the derivative/evaluation stack already supports multi-axis derivatives.

### Already Supported Internally

- Multi-dimensional surrogates `u(x,t)` or `u(x,y,z)`
- Derivatives along arbitrary axes
- Mixed derivatives (for example `u_xt`)
- Multi-output system discovery for coupled fields

### Remaining Work for General PDE Search

1. Generalize library generation beyond a single anchor axis.
2. Add first/second derivatives across all axes (`u_t`, `u_x`, `u_xx`, `u_tt`, `u_xt`, ...).
3. Support equation forms with multiple anchors (`u_tt - c^2 u_xx = 0`).
4. Expand model selection and sparsity penalties for richer PDE libraries.

Target equations include heat, wave, Burgers, and reaction-diffusion families.

---

## Dimensional Analysis

NestyNet\_SR optionally enforces dimensional consistency at two levels.

### Level 1: Local Consistency Checks

Four local gates validate individual operations:
- **Addition**: operands must have matching dimensions
- **Multiplication**: dimension exponents add
- **Exponentiation**: dimensions scale by the rational exponent
- **Transcendentals** (log, exp, sin, cos): argument must be dimensionless

### Level 2: Global Constraint Propagation (Buckingham-Sudoku)

All dimensional constraints are linear over ℚ. The full system is solved exactly via RREF in one pass, computing a per-node feasible dimension subspace (`DimSubspace`). Dead-end configurations are detected before any fitting.

**Hypothesis testing**: Compound and split proposals are tested non-mutatingly — build a candidate AST, solve from scratch, check for empty feasible sets. ~1 ms for 20 hypotheses.

```python
from nestynet_sr.sr_core.units import compute_node_domains, propose_split

# Compute per-node feasible dimensions
domains = compute_node_domains(root_ast, units_spec)  # None if infeasible

# Test a split hypothesis
result = propose_split(root, spec, atom, "add", group1=[0,1], group2=[2,3])
```

Dimensional analysis is enabled by default when a units spec is provided. Pass `--ignore_units` to disable.

---

## Generalized Symmetries

The generalized-symmetry (GS) layer (`nestynet_sr/sr_gs/`) discovers internal
coordinates from surrogate gradients through a learned general-affine
determining operator, going beyond any fixed detector menu: named-generator
audits (translations, scalings, rotations, boosts), arbitrary-real "oblique"
covectors, chart compositions (`identity`, `log`, `reciprocal`, `warp`), a
shared recursive carrier bank, and noise-calibrated promotion gates that
abstain rather than certify from noisy gradients.

- On `nestynet-sr` the GS Stage-A layer is **on by default** in propose mode
  (disable with `--gs-no-stagea`, or restrict to `--gs-mode audit`).
- On `nestynet-de` the layer is opt-in via `--gs-enable`, with an extensive
  `--gs-de-*` surface for ODE point symmetries and invariant compilation.
- The oracle FSS benchmark consumes GS carrier proposals via
  `--gs-carrier-seed` (see `examples/oracle_factorized_search/README.md`).
- Reports are written to `results/<stem>.gs_report.{json,md}`.

See `docs/source/user_guide/generalized_symmetries.rst` and
`docs/source/user_guide/nonlinear_de_symmetries.rst` for the full guide.

---

## Statistical Selection, Blinded Mode & Final Polish

Statistical selection (`nestynet_sr/stat_selection/`) produces a certified
selection report behind a search-vs-audit data firewall: audit rows are never
seen by the search, and the certificate records what the audit supports
(`--stat_certificate_json`, `--stat_archive_json`; full guide in
`docs/statistical_selection.md`).

Blinded mode (`--blinded`) forbids answer-key access during benchmark runs;
score afterwards with `scripts/score_blinded_run.py`.

The final Pareto polish (`run_sr_final_polish.py`, `--final_polish*` flags, or
the `nestynet-polish` console script) refits and prunes the leading candidates
into an accuracy-complexity slate under `results/<stem>_polish/`.

---

## Worked Examples & Paper Vignettes

The `examples/` tree carries the worked examples behind the NestyNet papers
(index in `examples/README.md`):

| Example | What it shows |
|---------|---------------|
| `kepler_ephemeris_real/` | Reduced Kepler hierarchy from 30 yr of JPL ephemerides for 308 asteroids: analytic surrogate accelerations, deterministic leverage split, Noether scan |
| `sparc_carrier/` | Blind recovery of the baryonic acceleration coordinate from SPARC mass models (Paper III vignette) |
| `jacobi_tidal/` | Galactic tidal-radius worked example with standalone note: GS discovery of the anisotropic invariant and principled noise abstention |
| `poisson_geometry/` | Certified Poisson-structure and Casimir discovery |
| `quadratic_symmetry/` | Bounded quadratic ODE point symmetries |
| `gs_charts/` | Graph symmetries compiled into executable charts (blast wave, SN 1993J) |
| `oracle_factorized_search/` | FSS oracle benchmark, including the GS carrier-seed bridge |

Benchmark reproduction is documented in `PAPER3_REPRODUCIBILITY.md` and
`PAPER4_REPRODUCIBILITY.md`.

---

## Directory Structure

```
nestynet_sr/
├── sr_core/               # Core AST nodes, units, separability
│   ├── bridges.py         # AST node types (Node union)
│   ├── units.py           # Dimensional analysis (Level 1 + Level 2)
│   ├── separability_math.py  # Separability detection
│   └── atoms.py           # Atom type definitions
├── sr_search/             # Search orchestration
│   ├── search.py          # Main search loop
│   ├── stageB/            # Stage B rewrite engine
│   │   ├── rules*.py      # 40+ rewrite rules
│   │   ├── engine.py      # Rule dispatcher
│   │   ├── main.py        # Pattern matchers
│   │   └── helpers.py     # Re-export hub
│   ├── factorized_search/           # factorized symbolic search explorer
│   ├── candidate_builders.py  # Compound proposals
│   └── compound_functions.py  # Compound function macros
├── sr_de/                 # Differential equation discovery
│   ├── de_search.py       # DE discovery pipeline
│   ├── system_de_search.py  # System/vector DE
│   ├── vector_ops.py      # Vector calculus macros (curl, div, grad, …)
│   ├── complex_ops.py     # Complex DE helpers
│   ├── de_templates.py    # VarPro template families
│   └── varpro_de.py       # VarPro Phase 1 & 2
├── adaptors/              # NestyNet optimization adaptors
│   ├── ast_composite.py   # ASTComposite adaptor
│   ├── de_varpro_adaptor.py
│   ├── template_varpro_base.py
│   └── template_varpro_adaptor.py
├── run_SR.py              # CLI entry point (nestynet-sr)
├── run_de.py              # CLI entry point (nestynet-de)
└── gui_sr.py              # Streamlit GUI

tests/                     # Test scripts (standalone + pytest-compatible)
examples/                  # Worked examples (see examples/README.md)
docs/                      # Sphinx documentation + statistical_selection.md
scripts/                   # Benchmark drivers and reproduction utilities
data/                      # Benchmark specs and generators
results/                   # Output files
```

---

## Output Files

Results are written to `results/`:

| File | Content |
|------|---------|
| `<stem>.human` | Human-readable discovered expression |
| `<stem>.state.pkl` | Checkpoint for resuming (SR only) |
| `<stem>.report.json` | Detailed JSON report with metrics |
| `<stem>_classSR.json` | Class-SR summary (shared/per-dataset tags + fitted params) |
| `<stem>_classSR.human` | Human-readable Class-SR report |
| `<stem>_de.human` | Human-readable discovered DE (from `run_de.py` or `run_SR.py --discover_de`) |
| `<stem>_de.pkl` | DE payload with term/residual ASTs and coefficients (from `run_SR.py --discover_de`) |
| `<stem>_de.json` | DE JSON report (from `run_de.py --save_json`) |
| `<stem>_final.human` | Final human-readable expression after Stage B |
| `<stem>_stageB.pkl` | Stage-B state payload |
| `<stem>.expressions.pkl` | Candidate expression archive (reloadable via `--load_expressions`) |
| `<stem>.gs_report.json` / `.gs_report.md` | Generalized-symmetry audit and proposal reports |
| `<stem>_polish/` | Final Pareto-polish outputs (accuracy-complexity slate) |
| statistical-selection certificate | Written where `--stat_certificate_json` / `--stat_archive_json` point |

---

## Testing

Tests are standalone scripts in `tests/` and `examples/`:

```bash
# Symbolic regression examples
python examples/logistic_growth/smoke_logistic_discovery.py
python examples/lane_emden/smoke_lane_emden_discovery.py

# Class-SR examples
python examples/classSR/smoke_quadratic_class.py
python examples/classSR/smoke_class_sr.py

# DE via direct and SR-first workflows
python examples/dho/smoke_dho_discovery.py --generate
python examples/dho/smoke_dho_discovery_sr.py --generate

# Adjoint symmetry
python tests/test_ast_composite_jvp_vjp_adjoint_audit.py

# Vector / System DE
python tests/test_wave_equation.py
python tests/test_coupled_1d_maxwell.py
python tests/test_maxwell_3d.py
python tests/test_navier_stokes_taylor_green.py

# Hamiltonian discovery
python tests/hamiltonian/test_hamiltonian_sho.py
python tests/hamiltonian/test_hamiltonian_multi.py

# Complex DE
python tests/test_complex_de_hello_world.py
python tests/test_complex_ops.py

# AI Feynman benchmark suite
python nestynet_sr/run_stageA_suite.py --only pb010
```

### Linting

```bash
ruff check nestynet_sr/
ruff format nestynet_sr/
mypy nestynet_sr/
```

> **Warning**: `ruff check --fix` can remove "unused" imports from re-export hubs (`stageB/helpers.py`, `stageB/__init__.py`). Always audit these files after auto-fix.

---

## License & Citation

**License**: Mozilla Public License 2.0 (MPL-2.0)

```bibtex
@software{nestynet_sr,
  author = {Ibata, Rodrigo},
  title = {NestyNet_SR: Symbolic Regression with Neural Networks},
  year = {2025}
}
```

---

## References

- **Symbolic Regression**: Separability detection in neural networks with analytic derivatives
- **DE Discovery (STLSQ)**: Brunton et al., "Discovering governing equations from data by sparse identification of nonlinear dynamical systems", PNAS 2016
- **PDE-FIND**: Rudy et al., "Data-driven discovery of partial differential equations", 2017
- **Variable Projection**: Golub & Pereyra, 1973; exploits separability in nonlinear least squares
- **NestyNet**: Segmented neural networks with analytic derivatives (10–100× faster than autograd)
- **Complex DE Examples**: `tests/test_complex_de_hello_world.py`, `tests/test_complex_ops.py`, `tests/test_complex_nodes.py`
- **Project docs**: this consolidated README plus `docs/source/` user guides and API docs
