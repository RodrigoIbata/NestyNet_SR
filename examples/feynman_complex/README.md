# Feynman Complex-Valued DE Benchmark

A benchmark suite of 26 complex-valued differential equations from physics, spanning quantum mechanics, nonlinear optics, classical oscillators, and condensed matter theory. Each problem is defined by a complex field (or system of complex fields) satisfying a known DE, and the active benchmark path rediscovers the coupled-real equations by building feature tables and running factorized symbolic search per equation.

## Quick Start

```bash
# Run the 3 default starter problems
python examples/feynman_complex/run_benchmark.py --fast

# Run a specific problem
python examples/feynman_complex/run_benchmark.py --only C001 --fast

# Run the full 26-problem benchmark
python examples/feynman_complex/run_benchmark.py --all --fast

# Full training (slower, more accurate)
python examples/feynman_complex/run_benchmark.py --all
```

## Problem Catalogue

### Quantum Mechanics (C000--C008)

| ID | Description | Equation | Type | Order |
|----|-------------|----------|------|-------|
| C000 | Free-particle Schrödinger | `iℏ∂ψ/∂t = -(ℏ²/2m)∂²ψ/∂x²` | PDE | 2 |
| C001 | Harmonic oscillator Schrödinger | `iℏ∂ψ/∂t = -(ℏ²/2m)∂²ψ/∂x² + ½mω²x²ψ` | PDE | 2 |
| C002 | 1D Schrödinger with potential | `iℏ∂ψ/∂t = -(ℏ²/2m)∂²ψ/∂x² + Vψ` | PDE | 2 |
| C003 | Gross-Pitaevskii (nonlinear Schrödinger) | `i∂ψ/∂t = -∂²ψ/∂x² + g\|ψ\|²ψ` | PDE | 2 |
| C004 | Two-level quantum system | `id[c₁,c₂]/dt = H·[c₁,c₂]` | ODE system | 1 |
| C005 | Pauli spin-1/2 equation | 2-component spinor in magnetic field | ODE system | 1 |
| C006 | Time-independent Schrödinger eigenvalue | `-(ℏ²/2m)d²ψ/dx² + Vψ = Eψ` | ODE | 2 |
| C007 | Hydrogen radial Schrödinger | `-(ℏ²/2m)(R'' + (2/r)R') + V_eff·R = E·R` | ODE | 2 |
| C008 | Dirac equation (1+1D) | `iγ^μ∂_μψ = mψ` (2-component spinor) | PDE system | 1 |

### Nonlinear Waves & Optics (C100--C105)

| ID | Description | Equation | Type | Order |
|----|-------------|----------|------|-------|
| C100 | Nonlinear Schrödinger / fiber optics | `i∂u/∂z + ∂²u/∂t² + \|u\|²u = 0` | PDE | 2 |
| C101 | Complex Ginzburg-Landau | `∂u/∂t = (1+ia)∂²u/∂x² + u - (1+ib)\|u\|²u` | PDE | 2 |
| C102 | Manakov system (coupled NLS) | `i∂u/∂z + u_tt + (\|u\|²+\|v\|²)u = 0` (same for v) | PDE system | 2 |
| C103 | NLS soliton envelope | `i∂A/∂t + ∂²A/∂x² + 2\|A\|²A = 0` | PDE | 2 |
| C104 | Second harmonic generation | `i∂A₁/∂z = κ·A₁*·A₂`, `i∂A₂/∂z = κ·A₁²` | ODE system | 1 |
| C105 | Parametric amplification | `i∂A/∂z = κ·A*·exp(iΔz)` | ODE | 1 |

### Classical Oscillators & Circuits (C200--C204)

| ID | Description | Equation | Type | Order |
|----|-------------|----------|------|-------|
| C200 | Complex damped oscillator | `z'' + 2γz' + ω₀²z = 0` | ODE | 2 |
| C201 | Driven complex oscillator | `z'' + 2γz' + ω₀²z = F·exp(iΩt)` | ODE | 2 |
| C202 | Coupled complex modes | `ż₁ = iω₁z₁ + κz₂`, `ż₂ = iω₂z₂ + κz₁` | ODE system | 1 |
| C203 | Van der Pol complex amplitude | `dA/dt = (μ-\|A\|²)A + iωA` | ODE | 1 |
| C204 | RLC phasor equation | `iωLI + RI + I/(iωC) = V` | algebraic | 0 |

### Condensed Matter & Field Theory (C300--C305)

| ID | Description | Equation | Type | Order |
|----|-------------|----------|------|-------|
| C300 | BCS gap equation | `dΔ/dT = g(N₀-\|Δ\|²)Δ` (GL relaxation) | ODE | 1 |
| C301 | Landau order parameter (TDGL) | `∂Φ/∂t = -aΦ - b\|Φ\|²Φ + κ∂²Φ/∂x²` | PDE | 2 |
| C302 | Josephson junction | `dφ/dt = V`, `dV/dt = -sin(φ) - αV + I_ext` | ODE system | 1 |
| C303 | Stuart-Landau oscillator | `dA/dt = (μ+iω)A - (1+iβ)\|A\|²A` | ODE | 1 |
| C304 | Complex Klein-Gordon field | `∂²φ/∂t² - c²∂²φ/∂x² + m²φ = 0` | PDE | 2 |
| C305 | Coupled Bose-Einstein condensates | `i∂ψ₁/∂t = -ψ₁'' + g₁\|ψ₁\|²ψ₁ + κψ₂` (and v.v.) | PDE system | 2 |

## Architecture

### Files

- **`problem_defs.py`** -- All 26 problem definitions: parameters, initial conditions, RHS functions (ODEs), analytic data generators (PDEs), ground-truth builders, and dimensional metadata. Legacy AST-library metadata is still kept here for older harnesses.
- **`run_benchmark.py`** -- Active benchmark runner: data generation, coupled-real feature-table construction, factorized symbolic search discovery, and rollout/residual validation.
- **`nestynet_sr/sr_core/problem_dims.py`** -- Shared canonical units metadata and adapters used by both `examples/feynman_de` and `examples/feynman_complex`.

### Pipeline (per problem)

1. **Data generation** -- ODEs via `solve_ivp`, PDEs via analytic solutions or spectral method-of-lines.
2. **Feature-table construction** -- Split each complex field into coupled-real components, then build shared feature columns from component values, derivatives, invariant terms such as `|psi|^2`, legal trig carriers, and declared constants.
3. **factorized symbolic search discovery** -- Run factorized symbolic search per discovered equation over the shared feature table, using `var_dims` / `y_dims` when dimensional metadata is available.
4. **Validation** -- Validate ODE discoveries by rollout and PDE/algebraic discoveries by residual checks against the known benchmark law.

### Feature Families

The active runner enumerates feature families rather than hand-built per-problem STLSQ libraries:

- Component values and anchor-adjacent derivatives.
- Pairwise products between real components.
- Complex invariants such as `|psi|^2` and `|psi|^2 * u_j`.
- Non-autonomous carriers such as `sin(Delta*x)` and `cos(Delta*x)` when the argument can be made dimensionless.
- Declared physical constants as scalar feature columns.

Legacy AST-library metadata is still present in `problem_defs.py` for the older harnesses and for reference ground-truth structures, but it is not the active benchmark path in `run_benchmark.py`.

## Fast Mode Note

`--fast` is a smoke configuration with reduced data and search budgets. It is
useful for wiring checks and quick regressions, but benchmark quality and final
pass/fail outcomes should be judged with the full-budget run rather than the
smoke setting.

## Key Design Decisions

- **Real decomposition**: Every complex field ψ = u + iv is split into real components and discovered as a coupled-real system.
- **Shared canonical units metadata**: `examples/feynman_de` and `examples/feynman_complex` now both feed the same canonical metadata layer in `nestynet_sr/sr_core/problem_dims.py`, then adapt that into solver-specific units payloads.
- **Dimension-gated features**: Illegal trig arguments are dropped unless they can be parameterized into dimensionless carriers, and componentwise target dimensions are forwarded into factorized symbolic search.
- **Analytic data for PDEs**: PDE problems use exact analytic solutions (plane waves, eigenstates, spectral solvers) rather than noisy numerical PDE solves when possible.
- **Spectral method-of-lines**: Nonlinear PDEs (C003, C100, C101, C102, C301, C305) use FFT for spatial derivatives + RK45 time integration.
- **Anchor order override**: PDEs like Schrödinger (1st in time, 2nd in space) use `ANCHOR_ORDER` to separate the temporal anchor from the spatial feature families.
