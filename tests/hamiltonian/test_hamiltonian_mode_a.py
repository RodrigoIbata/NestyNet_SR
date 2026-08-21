# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Test Mode A (fully shared H) multi-dataset discovery."""

import torch
import numpy as np
from nestynet_sr.sr_de.hamiltonian_search import (
    discover_hamiltonian_from_data_multi,
    HamiltonianSearchConfig,
)


def generate_sho_data(n_samples, omega, seed):
    """Generate (z, ż) data for SHO with H = 0.5*(p^2 + ω^2*q^2)."""
    np.random.seed(seed)

    q0 = np.random.randn(n_samples) * 2
    p0 = np.random.randn(n_samples) * 2

    # For Mode A test, use identical physics (same ω) across datasets
    qdot = p0
    pdot = -omega**2 * q0

    z = np.column_stack([q0, p0])
    zdot = np.column_stack([qdot, pdot])

    return torch.tensor(z, dtype=torch.float32), torch.tensor(zdot, dtype=torch.float32)


def main():
    print("=" * 70)
    print("Test: Mode A (Fully Shared H) Multi-Dataset Discovery")
    print("=" * 70)

    # Generate 3 datasets with IDENTICAL physics (ω=1.0 for all)
    # Mode A should discover one shared H
    omega = 1.0
    dataset_ids = [f"exp{i+1}" for i in range(3)]

    print(f"\n1. Generating 3 datasets with IDENTICAL physics (ω={omega}):")
    z_trains = []
    zdot_trains = []

    for i in range(3):
        z, zdot = generate_sho_data(n_samples=300, omega=omega, seed=42 + i)
        z_trains.append(z)
        zdot_trains.append(zdot)
        print(f"   Dataset {i}: H = 0.5*p^2 + 0.5*q^2  (same for all)")

    # Configure search
    print("\n2. Configuring Mode A discovery...")
    cfg = HamiltonianSearchConfig(
        n_dof=1,
        max_q_power=2,
        max_p_power=2,
        mechanical_split=True,
        include_const=False,
        stlsq_lambda=1e-3,
    )

    # Discover with Mode A (fully shared)
    print("\n3. Running Mode A (fully shared H) discovery...")
    result_a = discover_hamiltonian_from_data_multi(
        z_trains, zdot_trains,
        cfg=cfg,
        dataset_ids=dataset_ids,
        mode="shared"  # Mode A
    )

    # Also run Mode B for comparison
    print("\n4. Running Mode B (group-STLSQ) discovery for comparison...")
    result_b = discover_hamiltonian_from_data_multi(
        z_trains, zdot_trains,
        cfg=cfg,
        dataset_ids=dataset_ids,
        mode="group"  # Mode B
    )

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS: MODE A (Fully Shared)")
    print("=" * 70)

    print(f"\nShared Hamiltonian: {result_a.format_hamiltonian_for_dataset(0)}")
    print("\nAll datasets share IDENTICAL coefficients:")
    for d in range(3):
        print(f"  Dataset {d}: RMS = {result_a.rms_train[d]:.6f}")
        coeffs_str = ", ".join([f"a_{i}={c:.4f}" for i, c in enumerate(result_a.coeffs[d])])
        print(f"             Coeffs: {coeffs_str}")

    print("\n" + "=" * 70)
    print("RESULTS: MODE B (Group-STLSQ) for comparison")
    print("=" * 70)

    print("\nShared support, dataset-specific coefficients:")
    for d in range(3):
        print(f"  Dataset {d}: {result_b.format_hamiltonian_for_dataset(d)}")
        print(f"             RMS = {result_b.rms_train[d]:.6f}")

    # Verification
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Check Mode A: all datasets should have identical coefficients
    print("\n1. Mode A: Verifying all datasets have identical coefficients...")
    all_identical = True
    for d in range(1, 3):
        diff = (result_a.coeffs[d] - result_a.coeffs[0]).abs().max().item()
        if diff < 1e-6:
            print(f"   ✓ Dataset {d} matches dataset 0 (max diff: {diff:.2e})")
        else:
            print(f"   ✗ Dataset {d} differs from dataset 0 (max diff: {diff:.2e})")
            all_identical = False

    if all_identical:
        print("\n✓ Mode A SUCCESS: All datasets share identical coefficients!")
    else:
        print("\n✗ Mode A FAILED: Coefficients should be identical")

    # Check Mode B: should also converge to similar coefficients since physics is identical
    print("\n2. Mode B: Checking if coefficients are similar (physics is identical)...")
    coeffs_similar = True
    for d in range(1, 3):
        diff = (result_b.coeffs[d] - result_b.coeffs[0]).abs().max().item()
        if diff < 0.1:  # Allow some variation in Mode B
            print(f"   ✓ Dataset {d} similar to dataset 0 (max diff: {diff:.2e})")
        else:
            print(f"   ⚠ Dataset {d} differs from dataset 0 (max diff: {diff:.2e})")
            coeffs_similar = False
    assert coeffs_similar, "Mode B coefficients should remain similar across identical physics"

    # Check expected values
    print("\n3. Checking expected coefficient values...")
    expected_q = 0.5
    expected_p = 0.5

    for i, c in enumerate(result_a.coeffs[0]):
        term_str = repr(result_a.term_asts[i])
        if 'x0 ** 2' in term_str:  # q^2
            if abs(c.item() - expected_q) < 0.05:
                print(f"   ✓ q^2 coefficient: {c.item():.4f} ≈ {expected_q}")
            else:
                print(f"   ✗ q^2 coefficient: {c.item():.4f} ≠ {expected_q}")
        elif 'x1 ** 2' in term_str:  # p^2
            if abs(c.item() - expected_p) < 0.05:
                print(f"   ✓ p^2 coefficient: {c.item():.4f} ≈ {expected_p}")
            else:
                print(f"   ✗ p^2 coefficient: {c.item():.4f} ≠ {expected_p}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
