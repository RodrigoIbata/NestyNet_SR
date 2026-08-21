# Hamiltonian Discovery Tests

This directory contains tests for the Hamiltonian discovery functionality in NestyNet_SR.

## Overview

Hamiltonian discovery finds sparse Hamiltonian systems of the form `ż = J∇H(z)` from phase-space data `(z, ż)`, where:
- `z = (q, p)` is the phase-space state (positions and momenta/velocities)
- `J` is the symplectic matrix
- `H(z)` is discovered as a sparse linear combination: `H(z) = Σ aₖ φₖ(z)`

## Test Files

### test_hamiltonian_sho.py
**Purpose**: Basic single-dataset Hamiltonian discovery test

Tests discovery on simple harmonic oscillator (SHO) data with known Hamiltonian:
```
H = 0.5*q² + 0.5*p²
```

**What it validates**:
- Basic library term construction
- STLSQ sparsification
- Coefficient accuracy
- Training and validation RMS
- Correct identification of q² and p² terms

**Expected result**: Discovers H ≈ 0.5*x0² + 0.5*x1² with RMS < 1e-5

---

### test_hamiltonian_multi.py
**Purpose**: Multi-dataset discovery with shared support (Mode B / group-STLSQ)

Tests discovery across multiple SHO datasets with varying frequencies (ω = 1.0, 1.5, 2.0):
```
Dataset d: H_d = 0.5*ωd²*q² + 0.5*ωd²*p²
```

**What it validates**:
- Multi-dataset API (`discover_hamiltonian_from_data_multi`)
- Group-STLSQ: shared term support, dataset-specific coefficients
- Coefficient matrix structure (D × K)
- Per-dataset Hamiltonian formatting
- Coefficient ordering and variation across datasets

**Expected result**: Same terms (q², p²) selected for all datasets, but coefficients scale with ω²

---

### test_hamiltonian_mode_a.py
**Purpose**: Multi-dataset discovery with fully shared H (Mode A)

Compares Mode A (fully shared Hamiltonian) vs Mode B (shared support, dataset-specific coefficients) on identical SHO datasets (all with ω = 1.0).

**What it validates**:
- Mode A API (`mode="shared"`)
- Coefficient replication across datasets
- Comparison with Mode B behavior
- RMS values should be similar when physics is identical

**Expected result**:
- Mode A: identical coefficients replicated across all datasets
- Mode B: similar but potentially slightly different coefficients per dataset
- Both modes discover the correct H = 0.5*q² + 0.5*p² structure

---

### test_hamiltonian_const_term.py
**Purpose**: Edge case - constant term handling

Tests Hamiltonian discovery with `include_const=True` to verify that constant terms (which have zero gradient) are handled correctly by the autograd machinery.

**What it validates**:
- Constant term in library with `torch.autograd.grad(..., allow_unused=True)`
- No crashes or NaN/Inf values in design matrix
- Constant term may or may not be selected (it's a gauge freedom for H)
- Core terms (q², p²) still discovered correctly

**Expected result**: No crashes, core Hamiltonian structure recovered regardless of constant term

---

### test_hamiltonian_units.py
**Purpose**: Units consistency checking with Hamiltonian discovery

Tests integration between the units framework (`nestynet_sr.sr_core.units`) and discovered Hamiltonian ASTs.

**What it validates**:
1. **Test 1 (Dimensionless)**: Dimensionless phase space passes units check
   - q, p, H all dimensionless
   - Should always pass

2. **Test 2 (Physical Units)**: Physical units scenario
   - q = [L] (position)
   - p = [L·T⁻¹] (velocity)
   - H = [L²·T⁻²] (specific energy)
   - Units framework verifies dimensional consistency of AST structure

3. **Test 3 (Uniform Scaling)**: Both q and p have same dimension [L]
   - q = [L], p = [L]
   - H = [L²]
   - q² and p² are commensurate (same dimension)

**Expected result**: All three tests pass, demonstrating that `check_units_ast()` correctly validates discovered Hamiltonian structures

---

## Running the Tests

Run individual tests:
```bash
python tests/hamiltonian/test_hamiltonian_sho.py
python tests/hamiltonian/test_hamiltonian_multi.py
python tests/hamiltonian/test_hamiltonian_mode_a.py
python tests/hamiltonian/test_hamiltonian_const_term.py
python tests/hamiltonian/test_hamiltonian_units.py
```

Run all tests:
```bash
for test in tests/hamiltonian/test_*.py; do
    echo "Running $test..."
    python "$test" || exit 1
done
```

## Test Data

All tests use synthetic simple harmonic oscillator (SHO) data generated via:
```python
def generate_sho_data(n_samples, omega, seed):
    q0 = random samples
    p0 = random samples
    qdot = omega * p0
    pdot = -omega * q0
    z = [q0, p0]
    zdot = [qdot, pdot]
```

The true dynamics are:
```
q̇ = ∂H/∂p = ω²p
ṗ = -∂H/∂q = -ω²q
```

For ω = 1.0, the Hamiltonian is H = 0.5*(q² + p²).

## Key Features Tested

- ✓ Single-dataset Hamiltonian discovery
- ✓ Multi-dataset discovery (Mode A: fully shared H)
- ✓ Multi-dataset discovery (Mode B: shared support, dataset-specific coefficients)
- ✓ Mechanical splitting (H = T(p) + V(q))
- ✓ STLSQ sparsification
- ✓ Design matrix construction via PyTorch autograd
- ✓ Constant term handling (edge case)
- ✓ Units consistency checking
- ✓ AST construction and formatting
- ✓ Coefficient canonicalization

## Dependencies

Tests require:
- `torch` (PyTorch)
- `numpy`
- `nestynet_sr.sr_de.hamiltonian_search`
- `nestynet_sr.sr_core.units` (for units test only)
- `nestynet_sr.sr_core.bridges` (AST infrastructure)

## Notes

- Tests are standalone Python scripts (no pytest framework currently)
- All tests should pass and print success messages
- RMS values should be < 1e-5 for SHO discovery
- Tests document both expected behavior and edge cases
