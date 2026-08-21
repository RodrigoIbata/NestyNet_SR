# Maxwell Vector PDE Discovery

This example rediscovers the coupled vacuum Maxwell curl equations from synthetic field data:

```
dE/dt = +curl(B)
dB/dt = -curl(E)
```

It uses a plane-wave solution:

```
E = (sin(k*z - omega*t), 0, 0)
B = (0, sin(k*z - omega*t), 0)
omega = c*k
```

## What This Example Demonstrates

- Coupled multi-equation vector PDE discovery with `discover_vector_system_de_from_surrogate`.
- Maxwell-shaped candidate terms (`curl(B)`, `curl(E)`) with decoy terms (`E`, `B`).
- Explicit term-identity checks, not just residual checks.
- Data-first workflow:
  - generate tabulated fake fields and gradients,
  - build a lookup surrogate from those tables,
  - recover the equations.

## Run

From repository root:

```bash
# 1) Generate synthetic tabulated fields
python examples/Maxwell/generate_fake_maxwell_data.py

# 2) Recover Maxwell equations
python examples/Maxwell/discover_maxwell_from_fake_data.py

# 3) Plot fields and discovered residuals
python examples/Maxwell/plot_maxwell_fields.py
```

## Key Files

- `examples/Maxwell/generate_fake_maxwell_data.py`
  - Writes `data/maxwell/fake_maxwell_plane_wave.npz` with `X`, `Y`, `G`.
- `examples/Maxwell/discover_maxwell_from_fake_data.py`
  - Loads the tabulated data, runs discovery, and asserts:
    - Ampere equation has `coeff(curl(B)) ~= -1`,
    - Faraday equation has `coeff(curl(E)) ~= +1`,
    - cross-curl/decoy coefficients are near zero.
- `examples/Maxwell/plot_maxwell_fields.py`
  - Creates a 3x3 figure with field maps, derivative/curl comparisons, and residual maps from recovered equations.

## Expected Output

You should see:

- very small RMS residuals per component,
- a recovered system equivalent to:

```
[Ampere]  dE/dt - curl(B) = 0
[Faraday] dB/dt + curl(E) = 0
```

and a saved figure at `examples/Maxwell/maxwell_fields_and_residuals.png`.

## More Physical Toy: Wire Source

There is also a source-driven toy inspired by an AC current-carrying wire core.
It uses a smooth, localized current density `Jz(x,y,t)` and discovers:

```
dE/dt - curl(B) + J = 0
dB/dt + curl(E)     = 0
```

Run it:

```bash
# 1) Generate wire-source toy data
python examples/Maxwell/generate_fake_maxwell_wire_data.py

# 2) Discover source-driven Maxwell system with nuisance operators
python examples/Maxwell/discover_maxwell_wire_data.py

# 3) Plot wire-style field structure (Jz, B circulation, Ez)
python examples/Maxwell/plot_maxwell_wire_fields.py
```

This variant includes nuisance terms in the library:
`E` and `B` in addition to the physically relevant `curl(E)`, `curl(B)`, and `J`.

## Conductive Medium Toy

A second physically meaningful case is a conductive medium (no explicit source):

```
dE/dt - curl(B) + sigma*E = 0
dB/dt + curl(E)           = 0
```

The synthetic data uses exact damped plane-wave superposition (3 orthogonal modes),
so the damping is set by `sigma`.

Run it:

```bash
# 1) Generate conductive-medium data
python examples/Maxwell/generate_fake_maxwell_conductive_data.py

# 2) Discover conductive Maxwell equations with nuisance operators
python examples/Maxwell/discover_maxwell_conductive_data.py

# 3) Plot damped field behavior and RMS decay
python examples/Maxwell/plot_maxwell_conductive_fields.py
```

The discovery script checks:
- `coeff(curl(B)) ~= -1` in the E equation,
- `coeff(E) ~= +sigma` in the E equation,
- `coeff(curl(E)) ~= +1` in the B equation,
- nuisance terms near zero.
