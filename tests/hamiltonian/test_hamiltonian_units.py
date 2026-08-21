# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Test units consistency with Hamiltonian discovery.

This test verifies that:
1. Dimensionless Hamiltonian discovery passes units checks
2. Physical units are handled correctly in the discovered AST structure
3. Units framework correctly validates/rejects Hamiltonians based on dimensional analysis
"""

import torch
import numpy as np
from nestynet_sr.sr_de.hamiltonian_search import (
    discover_hamiltonian_from_data,
    HamiltonianSearchConfig,
)
from nestynet_sr.sr_core.units import (
    UnitSystem,
    UnitsSpec,
    check_units_ast,
)
from nestynet_sr.sr_core.bridges import collect_all_atoms


def _free_const_dims_for_ast(ast, dim):
    """Declare all trainable scalar coefficients in a discovered AST."""
    dims = {}
    for atom in collect_all_atoms(ast):
        if str(getattr(atom, "kind", "")).lower() not in ("free_const", "freeconst", "free_constant"):
            continue
        name = atom.kwargs.get("name") if isinstance(atom.kwargs, dict) else None
        if name is None:
            name = getattr(atom, "tag", None)
        if name is not None:
            dims[str(name)] = dim
    return dims


def generate_sho_data(n_samples=500, omega=1.0, seed=42):
    """Generate (z, ż) data for simple harmonic oscillator."""
    np.random.seed(seed)

    # Sample initial conditions
    q0 = np.random.randn(n_samples) * 2
    p0 = np.random.randn(n_samples) * 2

    # Dynamics: q̇ = ωp, ṗ = -ωq
    qdot = omega * p0
    pdot = -omega * q0

    # Stack into (N, 2) arrays
    z = np.column_stack([q0, p0])
    zdot = np.column_stack([qdot, pdot])

    return torch.tensor(z, dtype=torch.float32), torch.tensor(zdot, dtype=torch.float32)


def test_dimensionless_hamiltonian():
    """Test 1: Dimensionless phase space should always pass units check."""
    print("=" * 70)
    print("TEST 1: Dimensionless Hamiltonian Discovery")
    print("=" * 70)

    # Generate data
    print("\n1. Generating SHO data...")
    z_train, zdot_train = generate_sho_data(n_samples=500, omega=1.0)
    z_val, zdot_val = generate_sho_data(n_samples=200, omega=1.0, seed=123)

    # Configure search
    print("2. Discovering Hamiltonian...")
    cfg = HamiltonianSearchConfig(
        n_dof=1,
        max_q_power=2,
        max_p_power=2,
        mechanical_split=True,
        include_const=False,
        stlsq_lambda=1e-3,
    )

    result = discover_hamiltonian_from_data(
        z_train, zdot_train,
        z_val, zdot_val,
        cfg=cfg
    )

    print(f"\nDiscovered: {result.format_hamiltonian()}")
    print(f"Training RMS: {result.rms_train:.6f}")

    # Test units consistency with dimensionless spec
    print("\n3. Checking units consistency (dimensionless)...")
    us = UnitSystem(base=("L", "M", "T"))

    # All phase-space variables are dimensionless
    x_dims = (us.dimless(), us.dimless())  # (q, p) both dimensionless
    y_dim = us.dimless()  # H is dimensionless

    spec = UnitsSpec(
        unit_system=us,
        x_dims=x_dims,
        y_dim=y_dim,
        y_transform_name="identity",
        free_const_dims=_free_const_dims_for_ast(result.H_ast, us.dimless()),
    )

    check_result = check_units_ast(result.H_ast, spec)

    print(f"   Units check: {'PASS' if check_result.ok else 'FAIL'}")
    if not check_result.ok:
        print(f"   Reason: {check_result.reason}")

    if check_result.ok:
        print("\n✓ SUCCESS: Dimensionless Hamiltonian passes units check!")
    else:
        print("\n✗ FAILED: Dimensionless Hamiltonian should always pass!")
        raise AssertionError(f"Units check failed: {check_result.reason}")


def test_physical_units_hamiltonian():
    """Test 2: Physical units - verify AST structure compatibility.

    For SHO with phase-space convention z = (q, p) where p = velocity:
    - q has units [L] (length)
    - p has units [L*T^-1] (velocity)
    - H should have units [L^2*T^-2] (specific energy, energy per unit mass)

    With H = a*q^2 + b*p^2:
    - a*q^2 contributes [a]*[L^2]
    - b*p^2 contributes [b]*[L^2*T^-2]

    For dimensional consistency, we need [a] = [T^-2] and [b] = [1] (dimensionless)

    However, the current Hamiltonian discovery treats coefficients as dimensionless
    numbers. This test checks whether the AST *structure* allows for dimensional
    consistency if we were to assign appropriate units to coefficients.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Physical Units Hamiltonian Discovery")
    print("=" * 70)

    # Generate data
    print("\n1. Generating SHO data...")
    z_train, zdot_train = generate_sho_data(n_samples=500, omega=1.0)
    z_val, zdot_val = generate_sho_data(n_samples=200, omega=1.0, seed=123)

    # Configure search
    print("2. Discovering Hamiltonian...")
    cfg = HamiltonianSearchConfig(
        n_dof=1,
        max_q_power=2,
        max_p_power=2,
        mechanical_split=True,
        include_const=False,
        stlsq_lambda=1e-3,
    )

    result = discover_hamiltonian_from_data(
        z_train, zdot_train,
        z_val, zdot_val,
        cfg=cfg
    )

    print(f"\nDiscovered: {result.format_hamiltonian()}")
    print(f"Training RMS: {result.rms_train:.6f}")

    # Test with physical units: q=[L], p=[L*T^-1]
    print("\n3. Checking units consistency (physical units)...")
    print("   Phase space: q=[L], p=[L*T^-1]")
    print("   Expected H: [L^2*T^-2] (specific energy)")

    us = UnitSystem(base=("L", "M", "T"))

    # Phase-space dimensions
    x_dims = (
        us.dim([1, 0, 0]),      # q has units [L]
        us.dim([1, 0, -1]),     # p has units [L*T^-1]
    )
    y_dim = us.dim([2, 0, -2])  # H has units [L^2*T^-2]

    spec = UnitsSpec(
        unit_system=us,
        x_dims=x_dims,
        y_dim=y_dim,
        y_transform_name="identity",
        free_const_dims={
            "a0": us.dim([0, 0, -2]),
            "a1": us.dimless(),
        },
    )

    check_result = check_units_ast(result.H_ast, spec)

    print(f"   Units check: {'PASS' if check_result.ok else 'FAIL'}")
    if not check_result.ok:
        print(f"   Reason: {check_result.reason}")
        print("\n   EXPECTED BEHAVIOR: This may fail because discovered coefficients")
        print("   are treated as dimensionless. The AST structure (q^2 + p^2) cannot")
        print("   satisfy dimensional homogeneity without unit-carrying coefficients.")
        print("\n   For H = a*q^2 + b*p^2 to have units [L^2*T^-2]:")
        print("   - a*[L^2] = [L^2*T^-2]  =>  [a] = [T^-2]")
        print("   - b*[L^2*T^-2] = [L^2*T^-2]  =>  [b] = [1]")
        print("\n   Coefficients need different units for dimensional consistency.")
    else:
        print("\n✓ Units check PASSED!")
        print("   The units framework successfully validated the Hamiltonian structure.")

    assert check_result.ok, check_result.reason


def test_scaled_dimensionless():
    """Test 3: Verify that uniform scaling preserves units.

    If all phase-space variables have the same dimension [L], then q^2 and p^2
    both have dimension [L^2], making the sum dimensionally consistent.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Uniformly Scaled Phase Space")
    print("=" * 70)

    # Generate data
    print("\n1. Generating SHO data...")
    z_train, zdot_train = generate_sho_data(n_samples=500, omega=1.0)
    z_val, zdot_val = generate_sho_data(n_samples=200, omega=1.0, seed=123)

    # Configure search
    print("2. Discovering Hamiltonian...")
    cfg = HamiltonianSearchConfig(
        n_dof=1,
        max_q_power=2,
        max_p_power=2,
        mechanical_split=True,
        include_const=False,
        stlsq_lambda=1e-3,
    )

    result = discover_hamiltonian_from_data(
        z_train, zdot_train,
        z_val, zdot_val,
        cfg=cfg
    )

    print(f"\nDiscovered: {result.format_hamiltonian()}")

    # Test with uniform dimension: both q and p have units [L]
    print("\n3. Checking units consistency (uniform [L] scaling)...")
    print("   Phase space: q=[L], p=[L]")
    print("   Expected H: [L^2]")

    us = UnitSystem(base=("L", "M", "T"))

    # Both phase-space variables have dimension [L]
    x_dims = (
        us.dim([1, 0, 0]),      # q has units [L]
        us.dim([1, 0, 0]),      # p has units [L] (not physical, but mathematically valid)
    )
    y_dim = us.dim([2, 0, 0])  # H has units [L^2]

    spec = UnitsSpec(
        unit_system=us,
        x_dims=x_dims,
        y_dim=y_dim,
        y_transform_name="identity",
        free_const_dims=_free_const_dims_for_ast(result.H_ast, us.dimless()),
    )

    check_result = check_units_ast(result.H_ast, spec)

    print(f"   Units check: {'PASS' if check_result.ok else 'FAIL'}")
    if not check_result.ok:
        print(f"   Reason: {check_result.reason}")

    if check_result.ok:
        print("\n✓ SUCCESS: Uniformly scaled phase space passes units check!")
    else:
        print("\n✗ FAILED: Uniform scaling should allow dimensional consistency!")
        raise AssertionError(f"Units check failed: {check_result.reason}")


def main():
    print("\n" + "=" * 70)
    print("HAMILTONIAN DISCOVERY + UNITS CONSISTENCY TESTS")
    print("=" * 70)
    print("\nThese tests verify that the units framework can be applied to")
    print("discovered Hamiltonians to check dimensional consistency.\n")

    results = {}

    try:
        results['dimensionless'] = test_dimensionless_hamiltonian()
    except Exception as e:
        print(f"\n✗ Test 1 raised exception: {e}")
        results['dimensionless'] = False

    try:
        results['physical'] = test_physical_units_hamiltonian()
    except Exception as e:
        print(f"\n✗ Test 2 raised exception: {e}")
        results['physical'] = False

    try:
        results['uniform'] = test_scaled_dimensionless()
    except Exception as e:
        print(f"\n✗ Test 3 raised exception: {e}")
        results['uniform'] = False

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTest 1 (Dimensionless):  {'✓ PASS' if results['dimensionless'] else '✗ FAIL'}")
    print(f"Test 2 (Physical Units): {'✓ PASS' if results['physical'] else '✗ FAIL/EXPECTED'}")
    print(f"Test 3 (Uniform Scale):  {'✓ PASS' if results['uniform'] else '✗ FAIL'}")

    # Test 1 and 3 should pass, Test 2 may fail (documenting expected behavior)
    critical_pass = results['dimensionless'] and results['uniform']

    if critical_pass:
        print("\n✓ CRITICAL TESTS PASSED!")
        print("   The units framework successfully integrates with Hamiltonian discovery.")
        print("   Test 2 behavior is documented and expected.")
    else:
        print("\n✗ CRITICAL TEST FAILURE!")
        print("   The units framework has issues with Hamiltonian discovery.")
        raise AssertionError("Critical units tests failed")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
