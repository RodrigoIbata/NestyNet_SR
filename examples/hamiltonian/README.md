# Hamiltonian Discovery: Anharmonic Oscillator

Discovers the Hamiltonian of a 1-DOF anharmonic oscillator from phase-space trajectories:

```
H(q, p) = p^2/2 + q^2/2 + q^4/4
```

with dynamics `dq/dt = p`, `dp/dt = -q - q^3`.

## How It Works

For a Hamiltonian system `dz/dt = J * grad(H)`, the regression problem is **linear** in the unknown coefficients of `H` even though the dynamics are nonlinear. The script builds a monomial library `{q^2, q^4, p^2, ...}`, computes their symplectic gradients, and runs STLSQ to find the sparse active set.

The `mechanical_split=True` option enforces the physical constraint `H = T(p) + V(q)`, which halves the search space.

## Running

```bash
python examples/hamiltonian/anharmonic_oscillator.py
```

Runtime: a few seconds. The script generates training data (10 trajectories at different energies), runs discovery, validates coefficients, and saves plots.

## What to Expect

```
Discovered Hamiltonian:
  +0.5000 * q^2  +0.2500 * q^4  +0.5000 * p^2

Verification:
  q^2: found +0.5000, expected +0.5000, error < 0.01
  q^4: found +0.2500, expected +0.2500, error < 0.01
  p^2: found +0.5000, expected +0.5000, error < 0.01
```

## Output Files

Saved to `results/hamiltonian_anharmonic/`:

| File | Contents |
|------|----------|
| `phase_portrait.png` | Phase-space trajectories at different energies |
| `energy_conservation.png` | Time evolution and energy drift |
| `discovered_terms.png` | Bar plot of discovered vs expected coefficients |
