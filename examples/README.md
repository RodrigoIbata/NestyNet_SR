# Examples

This directory contains worked examples for symbolic regression and differential-equation discovery.

Run all commands from the repository root:

```bash
cd /path/to/NestyNet_SR
```

## Available Examples

| Example | Main Idea | Core Equation(s) | Entry Point |
|---------|-----------|------------------|-------------|
| [logistic_growth](logistic_growth/) | 1st-order ODE + template optimization | `du/dt = r*u*(1-u/K)` | `smoke_logistic_discovery.py` |
| [lane_emden](lane_emden/) | 2nd-order ODE + singular term | `d2y/dx2 + (2/x)*dy/dx + y^n = 0` | `smoke_lane_emden_discovery.py` |
| [dho](dho/) | 2nd-order damped oscillator from raw trajectory data (direct DE + SR-first DE) | `y'' + gamma*y' + omega^2*y = 0` | `smoke_dho_discovery.py`, `smoke_dho_discovery_sr.py` |
| [multi_dataset](multi_dataset/) | Shared-form discovery across datasets | Logistic family with varying `r` | `smoke_multi_logistic.py` |
| [hamiltonian](hamiltonian/) | Hamiltonian discovery from trajectories | `H = p^2/2 + q^2/2 + q^4/4` | `anharmonic_oscillator.py` |
| [poisson_geometry](poisson_geometry/) | General polynomial Poisson discovery with shared Hamiltonian heads, plus a Casimir taxonomy (physical Poisson Casimir vs full-rank algebra invariant, Hamiltonian gauge quotient) | quadratic cyclic Lotka--Volterra bracket; affine translated-Euler bracket; `so(3)*` Casimir vs `\|q x p\|^2` | `cyclic_lotka_volterra.py`, `translated_euler_top.py`, `casimir_taxonomy.py` |
| [kepler_ephemeris_real](kepler_ephemeris_real/) | Reduced-Kepler discovery staircase on real heliocentric ephemerides: analytic surrogate accelerations on data-discovered cylinder charts, 308-body ensemble with a deterministic 246/31/31 leverage split, six-panel showcase figure | `dot(theta)=h_d/r^2`, `ddot(r)=k_d/r^3-mu/r^2`, energy post-pass | `smoke_kepler_discovery.py`, `make_direct_paper_figures.py` |
| [Maxwell](Maxwell/) | Coupled vector PDE discovery | `dE/dt = curl(B)`, `dB/dt = -curl(E)` + source/conductive variants | `discover_maxwell_*.py` |
| [MOND](MOND/) | Nonlinear modified-Poisson benchmark | `div(mu(|grad(phi)|/a0) grad(phi)) = 4*pi*G*rho` | `run_benchmark.py` |
| [special_relativity](special_relativity/) | Operational interval discovery scaffold for Lorentzian kinematics | affine boost family, `r=-b/a=beta`, `1/a^2=1-beta^2`, invariant `u^2-x^2` | `smoke_interval_discovery.py` |
| [quadratic_symmetry](quadratic_symmetry/) | Nonlinear point-symmetry determining equations and invariant compilation | `u_xx=g/u^3`, special-conformal `x^2*d_x+x*u*d_u` | `conformal_inverse_square.py` |
| [oracle_factorized_search](oracle_factorized_search/) | Surrogate-free oracle harness for factorized symbolic search/continuous skeleton refinement with CLI + Streamlit GUIs (equation + DE modes) | User-defined equation specs and DE trajectory specs with dimensional constraints | `oracle_lab.py`, `oracle_lab_streamlit.py`, `oracle_lab_de_streamlit.py` |
| [feynman_de](feynman_de/) | Scalar DE benchmark used by Paper IV: 57 first/second-order ODEs from physics, multi-trajectory engines, declared-class `singular_origin` metadata keeping the term library answer-blind | Exponential decay, Lane-Emden, Bessel, driven/damped oscillators, ... | `run_benchmark.py` |
| [feynman_de_coe](feynman_de_coe/) | Repeatable detached launch scripts for overnight DE Committee-of-Experts validation runs over the scalar DE control cases | wraps `scripts/run_feynman_de_coe_control_suite.py` | `launch_full_adjudicate_detached.sh` |
| [feynman_complex](feynman_complex/) | 26-problem complex-valued DE benchmark (ODEs + PDEs, factorized symbolic search over coupled-real feature tables) | Schrödinger, NLS, Ginzburg-Landau, Dirac, Klein-Gordon, ... | `run_benchmark.py` |
| [classSR](classSR/) | Class-SR smoke runs through `run_SR.py`: shared symbolic form across related datasets (damped springs, quadratic families) | shared-form families with dataset-specific coefficients | `smoke_class_sr.py`, `smoke_quadratic_class.py` |
| [generalized_symmetries](generalized_symmetries/) | Entry points for the generalized-symmetry (GS) layer: analytic affine-generator audit and the Stage-A/DE smoke benchmark | rotation/scaling/translation invariants, e.g. `sin(sqrt(2)*x0-x1)` carrier discovery | `demo_affine_generators.py`, `gs_smoke_benchmark.py` |
| [gs_charts](gs_charts/) | GS -> charts bridge demos: continuous graph symmetries compiled into executable input charts, with blind Sedov-Taylor Trinity yield recovery and SN 1993J dating from real VLBI radii | `R = xi0 (E t^2/rho)^(1/5)`, power-law expansion with certificate | `demo_blast_wave.py`, `demo_sn1993j.py` |
| [gs_ablation](gs_ablation/) | Registry-driven baseline-vs-GS ablation runner for the examples tree (records commands, return codes, runtimes, GS reports) | orchestration only | `runner.py` |
| [jacobi_tidal](jacobi_tidal/) | Galactic tidal-radius vignette: GS discovery of the anisotropic tidal invariant and closed-form Jacobi-radius recovery, with a standalone note | `r_J = (mu/(4*Omega^2-kappa^2))^(1/3)` | `jacobi_tidal_note.pdf` (data, logs, figure shipped) |
| [sparc_carrier](sparc_carrier/) | SPARC baryonic-acceleration-carrier vignette for Paper III: blind discovery of the carrier `z = g_gas + Upsilon_d*g_disk` from real bulgeless-galaxy rotation curves | `g_obs = F(g_gas + Upsilon_d*g_disk)` | `build_dataset.py`, `run_pilot.py` |
| [FSS_figure](FSS_figure/) | Real-data figure explaining factorized-symbolic-search steering in SR and DE discovery; every panel is read from archived search reports or logs | figure-only | `make_paper_figures.py` |
| [core_acceptance_suites](core_acceptance_suites/) | JSON manifests driving `nestynet_sr/run_core_acceptance_suite.py`: explicit mathematical checks protecting the frozen SR/DE core | acceptance gates | `frozen_core_fast.json`, `frozen_core_smoke.json` |

## What This Collection Demonstrates

- End-to-end ODE/PDE discovery workflows from synthetic data.
- Sparse term selection with STLSQ / group-sparse STLSQ.
- Template-based refinement and LM-over-template-parameters (logistic/lane-emden).
- Coupled/vector equation discovery (Maxwell).
- Physics-structured discovery beyond scalar ODEs (Hamiltonian, Maxwell).
- Complex-valued DE discovery via real decomposition (ψ = u + iv), shared canonical units metadata, and factorized symbolic search over coupled-real feature tables (Feynman Complex).

## Quick Start By Example

### Logistic Growth

```bash
python examples/logistic_growth/generate_logistic_growth.py
python examples/logistic_growth/smoke_logistic_discovery.py --generate
python examples/logistic_growth/plot_results.py
```

### Lane-Emden

```bash
python examples/lane_emden/generate_lane_emden.py
python examples/lane_emden/smoke_lane_emden_discovery.py --generate
python examples/lane_emden/plot_results.py --n 1.0
```

### Damped Harmonic Oscillator (DHO)

```bash
python examples/dho/generate_dho.py
python examples/dho/smoke_dho_discovery.py --generate
python examples/dho/smoke_dho_discovery_sr.py --generate
```

### Multi-Dataset Logistic

```bash
python examples/multi_dataset/smoke_multi_logistic.py
```

### Hamiltonian (Anharmonic Oscillator)

```bash
python examples/hamiltonian/anharmonic_oscillator.py
```

### General Poisson Geometry

```bash
python -m examples.poisson_geometry.cyclic_lotka_volterra
python -m examples.poisson_geometry.translated_euler_top
```

### Kepler Ephemeris (Real)

```bash
python examples/kepler_ephemeris_real/generate_kepler_data.py
python examples/kepler_ephemeris_real/smoke_kepler_discovery.py --generate
python examples/kepler_ephemeris_real/smoke_kepler_discovery.py --generate --profile clean --enforce
```

Fast symbolic smoke on the real weathered profile:

```bash
python examples/kepler_ephemeris_real/run_class_sr_discovery.py --generate --fast --results_dir results/kepler_ephemeris_real_weathered_classsr_fast
```

### Maxwell

Base plane-wave toy:

```bash
python examples/Maxwell/generate_fake_maxwell_data.py
python examples/Maxwell/discover_maxwell_from_fake_data.py
python examples/Maxwell/plot_maxwell_fields.py
```

Wire-source toy:

```bash
python examples/Maxwell/generate_fake_maxwell_wire_data.py
python examples/Maxwell/discover_maxwell_wire_data.py
python examples/Maxwell/plot_maxwell_wire_fields.py
```

Conductive-medium toy:

```bash
python examples/Maxwell/generate_fake_maxwell_conductive_data.py
python examples/Maxwell/discover_maxwell_conductive_data.py
python examples/Maxwell/plot_maxwell_conductive_fields.py
```

### MOND

```bash
python examples/MOND/run_benchmark.py --only mond000 --fast
python examples/MOND/run_benchmark.py --all
```

### Special Relativity

```bash
python examples/special_relativity/generate_interval_data.py
python examples/special_relativity/smoke_interval_discovery.py --generate
```

### Quadratic Point Symmetry

```bash
python -m examples.quadratic_symmetry.conformal_inverse_square
```

### Oracle continuous skeleton refinement GUI

```bash
streamlit run examples/oracle_factorized_search/oracle_lab_streamlit.py
```

### Oracle DE continuous skeleton refinement GUI

```bash
streamlit run examples/oracle_factorized_search/oracle_lab_de_streamlit.py
```

For full oracle lab usage (CLI, suite runs, plotting), see `examples/oracle_factorized_search/README.md`.

### Feynman Complex DE Benchmark

```bash
# Run 3 starter problems
python examples/feynman_complex/run_benchmark.py --fast

# Run a specific problem
python examples/feynman_complex/run_benchmark.py --only C000 --fast

# Full 26-problem benchmark
python examples/feynman_complex/run_benchmark.py --all --fast
```

For the full problem catalogue (26 problems across quantum mechanics, nonlinear optics, classical oscillators, and condensed matter), see `examples/feynman_complex/README.md`.

## Validation Highlights

### Logistic + Lane-Emden

- Demonstrate three-way comparison: baseline STLSQ, heuristic template init, LM-optimized template parameters.
- Validate recovery quality through equation form, coefficients, and RMS residuals.

### Multi-Dataset

- Recovers shared support (`u`, `u^2`) with dataset-specific coefficients.

### Hamiltonian

- Recovers sparse Hamiltonian structure from phase-space trajectories.

### Kepler Ephemeris (Real)

- Runs the reduced-Kepler discovery staircase on heliocentric ephemeris trajectories, with an optional fully synthetic `astropy_builtin`/`clean` profile for controlled comparison.
- Recovers the areal law, shared inverse-square radial family, coefficient lift `k=h^2`, and reduced energy post-pass.
- Includes a symbolic Class-SR smoke runner that extracts inverse-power coefficient tables for `omega(r)` and `ddot(r)`.
- Supports a `clean` profile from real initial conditions plus exact two-body propagation, and a `weathered` profile from the ephemeris trajectory itself.
- Defaults to offline Astropy major-body ephemerides, but can ingest normalized external HORIZONS-like state CSVs.
- The clean direct-fit profile is exact to numerical precision; the weathered profile remains close to Kepler while exposing realistic departures.

### Maxwell

- Plane-wave case recovers curl coupling.
- Wire-source case recovers source-driven Ampere law (`+J`).
- Conductive case recovers damping term (`+sigma*E`) in Ampere law.

### MOND

- Uses manufactured potentials to produce reproducible nonlinear MOND datasets.
- Baseline sparse regression recovers the expected MOND feature support and coefficients.

### Special Relativity

- Generates operational 1+1D interval datasets across multiple inertial-frame pairs.
- Recovers per-regime affine boost matrices, then lifts them to the Lorentz coefficient laws.
- Extracts the preserved indefinite quadratic form, yielding the Minkowski interval in the current scaffold.

### Feynman Complex

- 26 complex-valued DEs from physics, split into coupled-real component systems.
- The active runner builds shared feature tables, forwards `var_dims` / `y_dims`, and runs factorized symbolic search per discovered equation.
- Shared canonical benchmark metadata now feeds both the scalar `feynman_de` and complex `feynman_complex` unit adapters.
- Covers ODEs (damped oscillators, quantum systems) and PDEs (Schrödinger, NLS, Klein-Gordon).

## Outputs

- Generated datasets go to `data/` (example-specific subfolders/files).
- Discovery outputs go to `results/`.
- Example-specific figures are saved inside each example directory (notably `examples/Maxwell/*.png`).

## Extension Ideas

- Damped oscillator with mixed sinusoidal/exponential structure.
- Van der Pol oscillator and other nonlinear limit-cycle systems.
- Additional PDE benchmarks (heat/reaction-diffusion, Navier-Stokes variants).
- More material models in Maxwell examples (anisotropy, dispersive media).
